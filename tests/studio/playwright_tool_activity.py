# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the collapse-tool-activity preference does to a rendered tool card.

The node suite reaches the two reducers in `tool-activity-open-state.ts` and the
preferences store, and asserts the JSX wiring against the TypeScript AST. None of
that can see a Radix Collapsible: whether closed content is in the DOM at all,
what `aria-expanded` says, or where the page ends up after a collapse. Those are
the questions here, against `studio/frontend/smoke-tool-activity.html`.

Scenes, each named for the claim it settles:

  1   baseline            preference off, tool running: every card open, as before the setting
  2   fresh mount         preference on: cards and group closed, the answer moves up
  3   manual expansion    survives live status and text updates (the first Codex P2)
  4   live toggle         a mounted, open card AND a mounted, open group both adopt the
                          preference (the second Codex P2, and the group asymmetry)
  5   persistence         the preference survives a reload and still collapses
  6   scroll              a close the preference causes must move the page exactly as far as
                          a close a click causes. Both arms are driven with the card already
                          in view, which is the whole trick: Playwright's click() scrolls its
                          target into view first, so an arm whose card is off-screen gets a
                          free scroll the other arm never gets, and the two then differ by
                          how they were driven rather than by what they ran.
  7   approval            with the preference on, a parked call still shows the command it
                          is asking approval for -- and collapses once approval is granted
  8   strict mode         the render-phase setState survives React's double render
  9   rtl                 the same collapse behaviour under dir=rtl
  10  reduced motion      prefers-reduced-motion must not leave a card stuck open
  11  toggle storm        rapid preference flips converge, with no render loop
  12  storage denied      localStorage throwing (Safari private browsing) must not take
                          the page down, and must land on the safe default
  13  malformed record    a hand-edited or foreign-written blob must not enable an opt-in

Engine follows the convention in playwright_settings_tabs.py:

    PW_ENGINE=chromium python tests/studio/playwright_tool_activity.py
    PW_ENGINE=firefox  python tests/studio/playwright_tool_activity.py --json
    PW_ENGINE=webkit   python tests/studio/playwright_tool_activity.py

chromium stands in for Chrome and Edge, webkit for Safari. Neither is the
branded browser; see the Playwright docs on that. It starts and stops its own
vite dev server -- point it at one you already have with SMOKE_BASE_URL, or move
the port it picks with SMOKE_PORT.

Exits non-zero if any scene misses its expectation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PAGE = "smoke-tool-activity.html"
ENTRY = "/smoke-tool-activity-main.tsx"
ENGINE = os.environ.get("PW_ENGINE", "chromium")
PORT = int(os.environ.get("SMOKE_PORT", "5219"))
PREFERENCES_KEY = "unsloth_chat_preferences"

# Radix animates for ANIMATION_DURATION = 200ms and the scroll lock holds for the
# same window, so every reading is taken well clear of both.
SETTLE_MS = 600

# The tail of the command the approval card renders. Deliberately past the
# 60-character slice the trigger shows, so "the user can read it" cannot be
# satisfied by the trigger label alone.
APPROVAL_TAIL = "--and-then-something-nobody-can-see"

CARDS = ("controlled", "uncontrolled", "approval")


def probe(page) -> dict:
    """Open-state of every disclosure, straight off the DOM.

    `aria-expanded` is what a screen reader gets and `data-state` drives what is
    painted, so both are recorded. `payload_in_dom` is separate on purpose: Radix
    keeps the content *wrapper* mounted while it animates, so a wrapper that
    exists says nothing about whether the text inside it does.
    """
    return page.evaluate(
        """(cards) => {
          const read = (name) => {
            const trigger = document.querySelector(`[data-probe="${name}-trigger"]`);
            const content = document.querySelector(`[data-probe="${name}-content"]`);
            return {
              aria_expanded: trigger ? trigger.getAttribute("aria-expanded") : null,
              data_state: content ? content.getAttribute("data-state") : null,
              height: content ? Math.round(content.getBoundingClientRect().height) : null,
              payload_in_dom: !!(content && content.querySelector("[data-probe]")),
              text_len: content ? content.innerText.length : 0,
            };
          };
          const out = {};
          for (const name of cards) out[name] = read(name);
          out.group = read("group");
          const answer = document.querySelector('[data-probe="answer"]');
          const viewport = document.querySelector('[data-probe="viewport"]');
          out.answer_top = answer ? Math.round(answer.getBoundingClientRect().top) : null;
          out.scroll_top = viewport ? Math.round(viewport.scrollTop) : null;
          out.approval_command_visible = (() => {
            const el = document.querySelector('[data-probe="approval-command"]');
            if (!el) return false;
            const rect = el.getBoundingClientRect();
            return rect.height > 0 && el.innerText.trim().length > 0;
          })();
          out.approval_command_text = (() => {
            const el = document.querySelector('[data-probe="approval-command"]');
            return el ? el.innerText : "";
          })();
          return out;
        }""",
        list(CARDS),
    )


def settle(page, ms: int = SETTLE_MS) -> None:
    page.wait_for_timeout(ms)


def open_page(
    page,
    base_url: str,
    *,
    fillers: int = 60,
    query: str = "",
) -> None:
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{base_url}/{PAGE}?fillers={fillers}{query}")
    try:
        page.wait_for_function("() => window.__probeReady", timeout = 30_000)
    except Exception:
        # A page that never mounts looks exactly like a slow one. Say which.
        raise RuntimeError(
            f"the harness page never published __probeReady ({ENGINE}). Page errors:\n"
            + ("\n".join(errors) or "(none captured)")
        ) from None
    settle(page)


def set_pref(page, value: bool) -> None:
    page.evaluate("(v) => window.__setPreference(v)", value)
    settle(page)


def fresh(page, base_url: str, preference: bool, **kwargs) -> None:
    """A page whose cards mounted with `preference` already in force."""
    open_page(page, base_url, **kwargs)
    page.evaluate("(v) => window.__setPreference(v)", preference)
    page.evaluate("() => window.__remount()")
    settle(page)


def run(base_url: str, pw) -> dict:
    results: dict = {"engine": ENGINE, "scenes": {}, "problems": []}
    p = results["problems"]
    s = results["scenes"]

    # `msedge` is not an engine, it is a chromium CHANNEL: the branded build
    # rather than Playwright's open-source one. Worth a pass of its own because
    # the branded builds trail Chromium by a few weeks, so a regression that
    # lands in Chromium first is invisible until Edge catches up.
    if ENGINE == "msedge":
        browser = pw.chromium.launch(channel = "msedge")
    else:
        browser = getattr(pw, ENGINE).launch()
    context = browser.new_context(viewport = {"width": 1200, "height": 900})
    page = context.new_page()
    console: list[str] = []
    page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))

    # --- 1 baseline -------------------------------------------------------
    fresh(page, base_url, False)
    s["1_baseline"] = probe(page)
    for card in CARDS:
        if s["1_baseline"][card]["aria_expanded"] != "true":
            p.append(f"1: {card} is not open with the preference OFF (baseline broken)")

    # --- 2 fresh mount ----------------------------------------------------
    fresh(page, base_url, True)
    s["2_fresh_mount"] = probe(page)
    for card in ("controlled", "uncontrolled"):
        if s["2_fresh_mount"][card]["aria_expanded"] == "true":
            p.append(f"2: {card} is still open with the preference ON")
    if s["2_fresh_mount"]["group"]["aria_expanded"] == "true":
        p.append("2: the group is open on a fresh mount with the preference ON")
    if s["2_fresh_mount"]["answer_top"] >= s["1_baseline"]["answer_top"]:
        p.append("2: the answer did not move up when activity collapsed")

    # --- 3 manual expansion ----------------------------------------------
    page.click('[data-probe="controlled-trigger"]')
    page.click('[data-probe="uncontrolled-trigger"]')
    settle(page)
    after_expand = probe(page)
    page.evaluate("() => window.__setHasText(true)")
    settle(page, 300)
    page.evaluate("() => window.__setRunning(false)")
    settle(page)
    after_updates = probe(page)
    s["3_manual_expansion"] = {"after_expand": after_expand, "after_updates": after_updates}
    for card in ("controlled", "uncontrolled"):
        if after_expand[card]["aria_expanded"] != "true":
            p.append(f"3: {card} did not open on click")
        elif after_updates[card]["aria_expanded"] != "true":
            p.append(f"3: {card} LOST its manual expansion across live updates")

    # --- 4 live toggle, cards and group ----------------------------------
    fresh(page, base_url, False)
    page.click('[data-probe="group-trigger"]')
    settle(page)
    before = probe(page)
    set_pref(page, True)
    after = probe(page)
    s["4_live_toggle"] = {"before": before, "after": after}
    for name in ("controlled", "uncontrolled", "group"):
        if before[name]["aria_expanded"] != "true":
            p.append(f"4: {name} was not open before the toggle")
        elif after[name]["aria_expanded"] == "true":
            p.append(f"4: {name} ignored the preference turning on while mounted")

    # --- 5 persistence ----------------------------------------------------
    stored = page.evaluate("(k) => window.localStorage.getItem(k)", PREFERENCES_KEY)
    open_page(page, base_url)
    scene = probe(page)
    scene["preference_after_reload"] = page.evaluate("() => window.__getPreference()")
    scene["stored_blob"] = stored
    s["5_persistence"] = scene
    if scene["preference_after_reload"] is not True:
        p.append("5: the preference did not survive a reload")
    for card in ("controlled", "uncontrolled"):
        if scene[card]["aria_expanded"] == "true":
            p.append(f"5: {card} is open after a reload with the preference stored")

    # --- 6 scroll ---------------------------------------------------------
    def close_one_card(via: str) -> tuple[int, int]:
        """Close ONE card, two ways, from an identical start.

        Two things make this a comparison rather than a coincidence. `only=uncontrolled`
        renders a single card, so both arms collapse the same content. And the card is
        scrolled INTO VIEW rather than the answer being centred: Playwright's click()
        scrolls its target into view before clicking, so centring the answer puts the card
        off-screen and hands the chevron arm a scroll the preference arm never gets. Driven
        that way the two arms appear to differ by ~665px; driven fairly they agree exactly.
        """
        fresh(page, base_url, False, query = "&only=uncontrolled")
        page.evaluate(
            """() => document.querySelector('[data-probe="uncontrolled-trigger"]')
                       .scrollIntoView({block: "start"})"""
        )
        settle(page)
        before = probe(page)
        if via == "preference":
            page.evaluate("() => window.__setPreference(true)")
        else:
            page.click('[data-probe="uncontrolled-trigger"]')
        settle(page)
        after = probe(page)
        return (
            after["answer_top"] - before["answer_top"],
            after["scroll_top"] - before["scroll_top"],
        )

    chevron = [close_one_card("chevron") for _ in range(3)]
    preference = [close_one_card("preference") for _ in range(3)]
    chevron_answer = statistics.median(m[0] for m in chevron)
    preference_answer = statistics.median(m[0] for m in preference)
    s["6_scroll"] = {
        "chevron": chevron,
        "preference": preference,
        "chevron_answer_median": chevron_answer,
        "preference_answer_median": preference_answer,
    }
    # A few pixels of slack for sub-pixel rounding; the two paths measure identical.
    if abs(preference_answer - chevron_answer) > 8:
        p.append(
            "6: a preference-driven close moves the page differently from a clicked close "
            f"({preference_answer}px vs {chevron_answer}px)"
        )
    for label, runs in (("chevron", chevron), ("preference", preference)):
        if any(m[1] != 0 for m in runs):
            p.append(f"6: a {label} close moved the scroll position: {[m[1] for m in runs]}")

    # --- 7 approval -------------------------------------------------------
    fresh(page, base_url, True)
    page.evaluate("() => window.__setAwaitingApproval(true)")
    settle(page)
    parked = probe(page)
    page.evaluate("() => window.__setAwaitingApproval(false)")
    settle(page)
    granted = probe(page)
    s["7_approval"] = {"parked": parked, "granted": granted}
    for card in ("approval", "controlled"):
        if parked[card]["aria_expanded"] != "true":
            p.append(f"7: the {card} card awaiting approval is collapsed by the preference")
    if not parked["approval_command_visible"]:
        p.append("7: the command being approved is not rendered")
    if APPROVAL_TAIL not in parked["approval_command_text"]:
        p.append(
            "7: only the truncated command is on screen -- 'Always allow' would be "
            "granted for text the user never saw"
        )
    for card in ("approval", "controlled"):
        if granted[card]["aria_expanded"] == "true":
            p.append(f"7: the {card} card stayed pinned open after approval was granted")

    # --- 8 strict mode ----------------------------------------------------
    fresh(page, base_url, True, query = "&strict=1")
    scene = probe(page)
    s["8_strict_mode"] = scene
    for card in ("controlled", "uncontrolled"):
        if scene[card]["aria_expanded"] == "true":
            p.append(f"8: {card} is open under StrictMode with the preference ON")
    loop_errors = [c for c in console if "Maximum update depth" in c or "Too many re-renders" in c]
    if loop_errors:
        p.append(f"8: React reported a render loop: {loop_errors[0]}")

    # --- 9 rtl ------------------------------------------------------------
    fresh(page, base_url, True, query = "&rtl=1")
    scene = probe(page)
    s["9_rtl"] = scene
    for card in ("controlled", "uncontrolled"):
        if scene[card]["aria_expanded"] == "true":
            p.append(f"9: {card} is open under dir=rtl with the preference ON")

    # --- 10 reduced motion ------------------------------------------------
    reduced = browser.new_context(viewport = {"width": 1200, "height": 900}, reduced_motion = "reduce")
    reduced_page = reduced.new_page()
    fresh(reduced_page, base_url, True)
    scene = probe(reduced_page)
    s["10_reduced_motion"] = scene
    for card in ("controlled", "uncontrolled"):
        if scene[card]["aria_expanded"] == "true":
            p.append(f"10: {card} is open under prefers-reduced-motion")
    reduced.close()

    # --- 11 toggle storm --------------------------------------------------
    fresh(page, base_url, False)
    page.evaluate(
        """() => {
          for (let i = 0; i < 40; i++) window.__setPreference(i % 2 === 1);
        }"""
    )
    settle(page)
    scene = probe(page)
    s["11_toggle_storm"] = scene
    if scene["controlled"]["aria_expanded"] != "false":
        p.append("11: the controlled card did not converge after 40 preference flips")
    if scene["uncontrolled"]["aria_expanded"] != "false":
        p.append("11: the uncontrolled card did not converge after 40 preference flips")

    # --- 12 storage denied -------------------------------------------------
    # Safari private browsing and a cookies-blocked profile both surface as
    # localStorage throwing rather than as an absent API.
    denied = browser.new_context(viewport = {"width": 1200, "height": 900})
    denied.add_init_script(
        """() => {
          const boom = () => { throw new DOMException("denied", "SecurityError"); };
          Object.defineProperty(window, "localStorage", {
            configurable: true,
            get: () => ({ getItem: boom, setItem: boom, removeItem: boom, clear: boom,
                          key: boom, length: 0 }),
          });
        }"""
    )
    denied_page = denied.new_page()
    denied_errors: list[str] = []
    denied_page.on("pageerror", lambda e: denied_errors.append(str(e)))
    try:
        open_page(denied_page, base_url)
        scene = probe(denied_page)
        scene["preference"] = denied_page.evaluate("() => window.__getPreference()")
        scene["declared_default"] = denied_page.evaluate("() => window.__getDefaultPreference()")
        scene["page_errors"] = denied_errors
        s["12_storage_denied"] = scene
        # Deliberately checked against the DECLARED default rather than against
        # a hard-coded value: which default ships is a product decision, and
        # this scene is about storage failing safe, not about which way.
        if scene["preference"] is not scene["declared_default"]:
            p.append(
                "12: a denied localStorage did not land on the declared default "
                f"({scene['preference']} vs {scene['declared_default']})"
            )
        if denied_errors:
            p.append(f"12: the page threw with localStorage denied: {denied_errors[0]}")
        expected = "false" if scene["declared_default"] else "true"
        for card in CARDS:
            if scene[card]["aria_expanded"] != expected:
                p.append(f"12: {card} does not match the declared default with storage denied")
    except RuntimeError as exc:
        s["12_storage_denied"] = {"failed": str(exc)}
        p.append(f"12: the page does not survive a denied localStorage: {exc}")
    denied.close()

    # --- 13 malformed record -----------------------------------------------
    malformed: dict = {}
    for label, raw in [
        ('string "false"', '"false"'),
        ('string "true"', '"true"'),
        ("number 1", "1"),
        ("number 0", "0"),
        ("empty string", '""'),
        ("object", "{}"),
        ("array", "[]"),
        ("null", "null"),
        ("corrupt json", None),
    ]:
        seeded = browser.new_context(viewport = {"width": 1200, "height": 900})
        blob = (
            "{not json"
            if raw is None
            else '{"state":{"collapseToolActivityByDefault":' + raw + '},"version":0}'
        )
        # add_init_script takes no arguments, so the values are baked into the
        # script rather than passed. json.dumps also handles the corrupt-JSON
        # case, which has to survive being embedded as a string literal.
        seeded.add_init_script(
            f"window.localStorage.setItem({json.dumps(PREFERENCES_KEY)}, {json.dumps(blob)});"
        )
        seeded_page = seeded.new_page()
        try:
            open_page(seeded_page, base_url)
            malformed[label] = seeded_page.evaluate("() => window.__getPreference()")
        except RuntimeError as exc:
            malformed[label] = f"page failed: {exc}"
            p.append(f"13: {label} takes the page down")
        seeded.close()
    s["13_malformed_record"] = malformed
    # Reported, not asserted: `?? false` accepts any non-nullish JSON value, and
    # that is true of every boolean in this store rather than of this one field.
    # It is not reachable from Studio's own UI. Recorded so a future change to
    # the store has a baseline to move.

    results["console"] = console
    browser.close()
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action = "store_true")
    options = ap.parse_args()

    base_url = os.environ.get("SMOKE_BASE_URL") or ""
    server = None
    if not base_url:
        server = start_vite(PORT)
        base_url = f"http://127.0.0.1:{PORT}"
        wait_for_smoke_page(
            f"{base_url}/{PAGE}",
            ENTRY,
            proc = server,
            info = lambda m: print(m, file = sys.stderr),
        )
    try:
        with sync_playwright() as pw:
            results = run(base_url, pw)
    finally:
        if server is not None:
            stop_process(server)

    if options.json:
        print(json.dumps(results, indent = 2))
    else:
        print(json.dumps(results["scenes"], indent = 2))

    print(file = sys.stderr)
    if results["problems"]:
        print(f"PROBLEMS ({ENGINE}):", file = sys.stderr)
        for problem in results["problems"]:
            print(f"  - {problem}", file = sys.stderr)
        return 1
    print(f"{ENGINE}: every scene matched its expectation", file = sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
