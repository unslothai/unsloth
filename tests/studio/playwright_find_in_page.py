# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Find in page, in a real browser, on three engines and three emulated platforms.

Drives smoke-find-in-page.html. The node suite covers the flatten, the offset
map and the search, which are pure; a Range has no geometry off a document,
CSS.highlights paints nothing, and a chord the browser owns cannot be taken
from it in a unit test, so the rest is only answerable here.

    SMOKE_ENGINES=chromium,firefox,webkit python3 tests/studio/playwright_find_in_page.py

Three engine capabilities decide which path the bar takes, each degraded
deliberately rather than waited for:

  - no Custom Highlight API (Firefox below 140, and whatever WebKitGTK the host
    shipped), so the bar falls back to selecting the active match;
  - checkVisibility honouring only the historic option names (Chrome 105-120,
    Firefox 106-121), where Web IDL drops the modern spellings silently and
    hidden text would otherwise be indexed;
  - no checkVisibility at all (Safari below 17.4, WebKitGTK), where the call
    answers undefined and the computed properties stand in.

Platform is emulated the way the app reads it, through navigator.platform and
the user agent, because isMacPlatform() is memoised on first call.
"""

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
)

PORT = int(os.environ.get("SMOKE_PORT", "5419"))
ENGINES = [e for e in os.environ.get("SMOKE_ENGINES", "chromium").split(",") if e]
URL = f"http://127.0.0.1:{PORT}/smoke-find-in-page.html"

PLATFORMS = {
    "macOS": ("MacIntel", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) SmokeUA"),
    "Windows": ("Win32", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SmokeUA"),
    "Linux": ("Linux x86_64", "Mozilla/5.0 (X11; Linux x86_64) SmokeUA"),
}
MOD = {"macOS": "Meta", "Windows": "Control", "Linux": "Control"}

# Drop the highlight registry before the app loads, so the bar takes the
# selection fallback exactly as it does on an engine that never had one.
NO_HIGHLIGHT_API = """
delete window.Highlight;
try { delete CSS.highlights; } catch (e) { /* getter-only on some engines */ }
"""

# An engine from before the option rename: it answers only checkVisibilityCSS
# and checkOpacity, and ignores the modern spellings the way Web IDL does.
LEGACY_CHECK_VISIBILITY = """
const real = Element.prototype.checkVisibility;
Element.prototype.checkVisibility = function (options) {
  const legacy = {};
  if (options && 'checkVisibilityCSS' in options) {
    legacy.checkVisibilityCSS = options.checkVisibilityCSS;
  }
  if (options && 'checkOpacity' in options) {
    legacy.checkOpacity = options.checkOpacity;
  }
  if (options && 'contentVisibilityAuto' in options) {
    legacy.contentVisibilityAuto = options.contentVisibilityAuto;
  }
  return real.call(this, legacy);
};
"""

# An engine with no checkVisibility at all, which is Safari below 17.4 and the
# WebKitGTK the desktop build is handed. The optional call answers undefined
# there, and read as "not false" it put every display: none subtree back in.
NO_CHECK_VISIBILITY = """
delete Element.prototype.checkVisibility;
"""

failures: list[str] = []
passed = 0


def check(
    engine: str,
    mode: str,
    name: str,
    ok: bool,
    detail: str = "",
) -> None:
    global passed
    if ok:
        passed += 1
        return
    failures.append(f"{engine}/{mode}: {name}\n      {detail}")


def counter(page) -> str | None:
    return page.evaluate("() => window.__findSmoke.counter()")


def state(page) -> dict:
    return page.evaluate("() => window.__findSmoke.store.getState()")


def open_bar(page, mod: str) -> None:
    page.keyboard.press(f"{mod}+f")
    page.wait_for_timeout(250)


def close_bar(page) -> None:
    page.keyboard.press("Escape")
    page.wait_for_timeout(250)


def check_chord(page, engine: str, mode: str, mod: str) -> None:
    """The chord opens the bar, and does so from a text field too."""
    open_bar(page, mod)
    check(engine, mode, "the chord opens the bar", state(page).get("open") is True)
    check(
        engine,
        mode,
        "the bar is a search landmark",
        page.locator('[role="search"]').count() == 1,
    )

    # Re-pressing it while the field has focus restarts the search rather than
    # handing the chord back to the browser.
    before = state(page).get("focusToken")
    page.keyboard.press(f"{mod}+f")
    page.wait_for_timeout(200)
    check(
        engine,
        mode,
        "the chord re-focuses the field instead of closing",
        state(page).get("open") is True and state(page).get("focusToken") != before,
    )
    close_bar(page)
    check(engine, mode, "Escape closes the bar", state(page).get("open") is False)


def check_counting(page, engine: str, mode: str, mod: str) -> None:
    open_bar(page, mod)
    page.keyboard.type("unsloth")
    page.wait_for_timeout(500)
    shown = counter(page)
    check(engine, mode, "a query is counted", bool(shown), f"counter={shown!r}")

    # An off-route workspace is kept mounted under `inert`; its matches must not
    # be counted or walked to.
    page.evaluate("() => window.__findSmoke.setWorkspace('other')")
    page.wait_for_timeout(700)
    other = counter(page)
    check(
        engine,
        mode,
        "an inert workspace is out of the count",
        other != shown,
        f"{shown!r} -> {other!r}",
    )
    page.evaluate("() => window.__findSmoke.setWorkspace('chat')")
    page.wait_for_timeout(700)
    check(
        engine,
        mode,
        "switching back restores the count",
        counter(page) == shown,
        f"{counter(page)!r} != {shown!r}",
    )
    close_bar(page)


def check_walk(page, engine: str, mode: str, mod: str) -> None:
    open_bar(page, mod)
    page.keyboard.type("unsloth")
    page.wait_for_timeout(500)
    start = page.evaluate("() => window.__findSmoke.scrollTop()")
    for _ in range(6):
        page.keyboard.press("Enter")
        page.wait_for_timeout(120)
    moved = page.evaluate("() => window.__findSmoke.scrollTop()")
    check(
        engine,
        mode,
        "Enter walks the matches and scrolls to them",
        moved != start,
        f"scrollTop {start} -> {moved}",
    )

    # Shift+Enter walks back.
    page.keyboard.press("Shift+Enter")
    page.wait_for_timeout(200)
    check(
        engine,
        mode,
        "Shift+Enter walks backwards",
        counter(page) is not None,
        f"counter={counter(page)!r}",
    )
    close_bar(page)


def check_streaming(page, engine: str, mode: str, mod: str) -> None:
    open_bar(page, mod)
    page.keyboard.type("unsloth")
    page.wait_for_timeout(500)
    for _ in range(4):
        page.keyboard.press("Enter")
        page.wait_for_timeout(120)
    before_scroll = page.evaluate("() => window.__findSmoke.scrollTop()")
    before_count = counter(page)
    page.evaluate("() => window.__findSmoke.stream('a streamed unsloth reply', 5)")
    page.wait_for_timeout(1600)
    check(
        engine,
        mode,
        "a streamed reply raises the count",
        counter(page) != before_count,
        f"{before_count!r} -> {counter(page)!r}",
    )
    check(
        engine,
        mode,
        "a streamed reply does not move the reader",
        page.evaluate("() => window.__findSmoke.scrollTop()") == before_scroll,
        f"scrollTop {before_scroll} -> " f"{page.evaluate('() => window.__findSmoke.scrollTop()')}",
    )
    close_bar(page)


def check_paint_and_teardown(page, engine: str, mode: str, mod: str) -> None:
    """Whichever way this engine shows a match, opening paints and closing clears."""
    has_api = page.evaluate("() => typeof CSS !== 'undefined' && !!CSS.highlights")
    open_bar(page, mod)
    page.keyboard.type("unsloth")
    page.wait_for_timeout(500)

    if has_api:
        painted = page.evaluate(
            "() => ({ all: CSS.highlights.has('unsloth-find'),"
            " active: CSS.highlights.has('unsloth-find-active') })"
        )
        check(
            engine,
            mode,
            "both highlight registries are painted",
            painted["all"] and painted["active"],
            str(painted),
        )
    else:
        # No registry, so the bar falls back to selecting the active match. Whether that selection
        # survives is the engine's call - Gecko keeps it, WebKit and Blink drop it to give the caret
        # back - so the invariant here is the one that is not negotiable: the field still works.
        # Moving the selection out from under a focused field used to swallow every keystroke after
        # the first, which froze the query at one character on exactly the engines that land here.
        typed = page.evaluate("() => document.querySelector('[role=\"search\"] input')?.value")
        check(
            engine,
            mode,
            "the field is still typable once the fallback has selected",
            typed == "unsloth",
            f"input value={typed!r}, expected 'unsloth'",
        )
        check(
            engine,
            mode,
            "the fallback still counts every match",
            bool(counter(page)),
            f"counter={counter(page)!r}",
        )

    close_bar(page)
    if has_api:
        left = page.evaluate(
            "() => CSS.highlights.has('unsloth-find')"
            " || CSS.highlights.has('unsloth-find-active')"
        )
        check(engine, mode, "closing clears both registries", left is False)
    else:
        check(
            engine,
            mode,
            "closing drops the fallback selection",
            page.evaluate("() => (window.getSelection()?.toString() ?? '')") == "",
        )


def check_modal_gate(page, engine: str, mode: str, mod: str) -> None:
    """A modal backgrounds the scope, and the chord must go back to the browser."""
    page.evaluate("() => window.__findSmoke.setModal(true)")
    page.wait_for_timeout(250)
    open_bar(page, mod)
    check(
        engine,
        mode,
        "the chord is declined while a modal backgrounds the scope",
        state(page).get("open") is False,
    )
    page.evaluate("() => window.__findSmoke.setModal(false)")
    page.wait_for_timeout(250)
    open_bar(page, mod)
    check(
        engine,
        mode,
        "the chord works again once the modal is gone",
        state(page).get("open") is True,
    )
    close_bar(page)


def check_hidden_text(page, engine: str, mode: str, mod: str) -> None:
    """Text nobody can see must not be counted, however it was hidden.

    `display: none` is the case an engine with no checkVisibility gets wrong,
    `visibility: hidden` is the case an engine from before the checkVisibility
    option rename gets wrong, and `display: contents` + `visibility: hidden` is
    the case where only ELEMENT children are re-checked, so a direct text child
    slips through. All three are planted here rather than in the harness so this
    stays honest about what it is measuring.
    """
    page.evaluate(
        """() => {
      const scope = document.querySelector('[data-find-scope]') ?? document.body;
      for (const id of ['probe-vis', 'probe-contents', 'probe-none']) {
        document.getElementById(id)?.remove();
      }
      const gone = document.createElement('div');
      gone.id = 'probe-none';
      gone.style.display = 'none';
      gone.textContent = 'zqxjkvbrmp';
      scope.appendChild(gone);

      const plain = document.createElement('div');
      plain.id = 'probe-vis';
      plain.style.visibility = 'hidden';
      plain.textContent = 'zqxjkvbrmp';
      scope.appendChild(plain);

      const wrapper = document.createElement('span');
      wrapper.id = 'probe-contents';
      wrapper.style.display = 'contents';
      wrapper.style.visibility = 'hidden';
      wrapper.textContent = 'zqxjkvbrmp';
      scope.appendChild(wrapper);
    }"""
    )
    page.wait_for_timeout(500)
    open_bar(page, mod)
    page.keyboard.type("zqxjkvbrmp")
    page.wait_for_timeout(700)
    shown = counter(page)
    check(
        engine,
        mode,
        "hidden text is not findable",
        shown in (None, "", "0/0"),
        f"counter={shown!r} (a non-zero count means hidden text was indexed)",
    )
    close_bar(page)
    page.evaluate(
        """() => {
      document.getElementById('probe-vis')?.remove();
      document.getElementById('probe-contents')?.remove();
      document.getElementById('probe-none')?.remove();
    }"""
    )


def check_content_visibility_reveal(page, engine: str, mode: str, mod: str) -> None:
    """A match under `content-visibility: auto` has to end up on screen.

    A scroll reaches only as far as the scrollHeight the engine knows about, and a skipped subtree
    contributes its `contain-intrinsic-size` placeholder until it renders. Reaching toward it is
    what makes it relevant, so the document grows underneath the scroll that was aimed at it. The
    Hub puts this containment on README prose and on each top-level child (hub.css), and only a real
    viewport can measure it, which is why it is here and not in the node suite.
    """
    planted = page.evaluate(
        """() => {
      const conversation = document.querySelector('[data-find-scope] .mx-auto');
      document.getElementById('cv-probe')?.remove();
      const box = document.createElement('div');
      box.id = 'cv-probe';
      box.style.contentVisibility = 'auto';
      box.style.containIntrinsicSize = 'auto 140px';
      const lines = [];
      for (let i = 0; i < 120; i++) lines.push('filler line ' + i);
      lines.push('zqxjcvneedle at the very bottom');
      box.innerHTML = lines.map((t) => '<p>' + t + '</p>').join('');
      conversation.appendChild(box);
      return Math.round(box.getBoundingClientRect().height);
    }"""
    )
    check(
        engine,
        mode,
        "the placeholder really is standing in for the block",
        planted == 140,
        f"placeholder height={planted}",
    )
    page.wait_for_timeout(600)
    open_bar(page, mod)
    # Inserted whole, the way a paste arrives, rather than typed. Typing hides this: every
    # keystroke reveals again, and those repeats do by accident what the fix does on purpose.
    # One change is what a paste, an Enter, or a click on the walk button each produce.
    page.keyboard.insert_text("zqxjcvneedle")
    page.wait_for_timeout(1200)
    where = page.evaluate(
        """() => {
      const p = [...document.querySelectorAll('#cv-probe p')]
        .find((n) => n.textContent.includes('zqxjcvneedle'));
      const r = p.getBoundingClientRect();
      return { top: Math.round(r.top), bottom: Math.round(r.bottom),
               height: window.innerHeight,
               real: Math.round(document.getElementById('cv-probe').getBoundingClientRect().height),
               inViewport: r.top >= 0 && r.bottom <= window.innerHeight };
    }"""
    )
    check(
        engine,
        mode,
        "the block behind the placeholder rendered",
        where["real"] > 1000,
        f"height={where['real']} (still the placeholder)",
    )
    check(
        engine,
        mode,
        "typing the query leaves the match on screen",
        where["inViewport"] is True,
        f"{where} (aimed at the placeholder and never looked again)",
    )
    close_bar(page)
    page.evaluate("() => document.getElementById('cv-probe')?.remove()")


def check_spelling_variants(page, engine: str, mode: str, mod: str) -> None:
    """A word composed and a word decomposed have to find each other.

    macOS hands back decomposed filenames while a model writes composed prose,
    so one thread holds both. The index is left in the form the document wrote,
    since normalizing it would change its length and misplace every offset, so
    what is checked here is that the painted range still covers exactly the
    characters that were written.
    """
    planted = page.evaluate(
        """() => {
      const scope = document.querySelector('[data-find-scope]') ?? document.body;
      document.getElementById('probe-nfd')?.remove();
      const row = document.createElement('div');
      row.id = 'probe-nfd';
      // cafe + combining acute, then " vbnmqz" to keep the query out of the prose.
      row.textContent = 'cafe\u0301 vbnmqz';
      scope.appendChild(row);
      return row.textContent.normalize('NFC') !== row.textContent;
    }"""
    )
    check(engine, mode, "the planted text really is decomposed", planted is True)
    page.wait_for_timeout(500)
    open_bar(page, mod)
    # Typed composed, against text written decomposed.
    page.keyboard.type("caf\u00e9 vbnmqz")
    page.wait_for_timeout(700)
    shown = counter(page)
    check(
        engine,
        mode,
        "a composed query finds decomposed text",
        shown == "1/1",
        f"counter={shown!r}",
    )
    # Only where there is a registry to read. On the fallback path the selection is dropped on
    # purpose to give the caret back to the field, so there is nothing left to measure; the count
    # above is the part that holds on every engine.
    if page.evaluate("() => typeof CSS !== 'undefined' && !!CSS.highlights"):
        covers = page.evaluate(
            """() => {
      const set = CSS.highlights.get('unsloth-find-active');
      const range = set ? [...set][0] : null;
      return range ? range.toString() : null;
    }"""
        )
        check(
            engine,
            mode,
            "the painted range covers what the document wrote",
            covers == "cafe\u0301 vbnmqz",
            f"painted={covers!r} (a drifted offset paints the wrong characters)",
        )
    close_bar(page)
    page.evaluate("() => document.getElementById('probe-nfd')?.remove()")


def check_skip_attribute(page, engine: str, mode: str, mod: str) -> None:
    """Marking a region skippable has to take it out of the count."""
    page.evaluate(
        """() => {
      const scope = document.querySelector('[data-find-scope]') ?? document.body;
      document.getElementById('probe-skip')?.remove();
      const panel = document.createElement('div');
      panel.id = 'probe-skip';
      panel.textContent = 'wqzlmxtvbn wqzlmxtvbn';
      scope.appendChild(panel);
    }"""
    )
    page.wait_for_timeout(500)
    open_bar(page, mod)
    page.keyboard.type("wqzlmxtvbn")
    page.wait_for_timeout(700)
    before = counter(page)
    check(engine, mode, "the planted panel is counted", bool(before), f"{before!r}")

    # Adding the attribute is the direction that used to be filtered out of the
    # observer's own batch, leaving the region counted until something else
    # happened to mutate.
    page.evaluate("() => document.getElementById('probe-skip')?.setAttribute('data-find-skip', '')")
    page.wait_for_timeout(900)
    after = counter(page)
    check(
        engine,
        mode,
        "marking a region data-find-skip reindexes and drops it",
        after != before,
        f"{before!r} -> {after!r} (unchanged means the mutation was filtered out)",
    )
    close_bar(page)
    page.evaluate("() => document.getElementById('probe-skip')?.remove()")


def run_page(page, engine: str, mode: str, mod: str) -> None:
    for probe in (
        check_chord,
        check_counting,
        check_walk,
        check_streaming,
        check_paint_and_teardown,
        check_modal_gate,
        check_hidden_text,
        check_content_visibility_reveal,
        check_spelling_variants,
        check_skip_attribute,
    ):
        try:
            probe(page, engine, mode, mod)
        except Exception as exc:  # noqa: BLE001
            check(
                engine,
                mode,
                f"{probe.__name__} ran to completion",
                False,
                f"{type(exc).__name__}: {str(exc)[:300]}",
            )


def new_page(context):
    page = context.new_page()
    for attempt in range(30):
        try:
            page.goto(URL, wait_until = "domcontentloaded", timeout = 30000)
            break
        except Exception:
            if attempt == 29:
                raise
            time.sleep(2)
    page.wait_for_function("() => !!window.__findSmoke", timeout = 120000)
    return page


def run_engine(pw, engine: str) -> None:
    launch = {"args": chromium_launch_args()} if engine == "chromium" else {}
    browser = getattr(pw, engine).launch(**launch)
    try:
        # The platform sweep, on the engine's own capabilities.
        for platform, (nav_platform, agent) in PLATFORMS.items():
            context = browser.new_context(user_agent = agent)
            context.add_init_script(
                "Object.defineProperty(navigator, 'platform', "
                f"{{ get: () => {json.dumps(nav_platform)} }});"
            )
            page = new_page(context)
            run_page(page, engine, platform, MOD[platform])
            context.close()

        # The two degraded engines, on one platform each: what they exercise is
        # the capability, and the platform sweep above already covered the chord.
        for mode, script in (
            ("no-highlight-api", NO_HIGHLIGHT_API),
            ("legacy-checkVisibility", LEGACY_CHECK_VISIBILITY),
            ("no-checkVisibility", NO_CHECK_VISIBILITY),
        ):
            context = browser.new_context(user_agent = PLATFORMS["Linux"][1])
            context.add_init_script(
                "Object.defineProperty(navigator, 'platform', { get: () => 'Linux x86_64' });"
            )
            context.add_init_script(script)
            page = new_page(context)
            run_page(page, engine, mode, "Control")
            context.close()
    finally:
        browser.close()


def main() -> int:
    server = start_vite(PORT)
    try:
        with sync_playwright() as pw:
            for engine in ENGINES:
                run_engine(pw, engine)
    finally:
        stop_process(server)

    total = passed + len(failures)
    print(f"[find-in-page] {passed}/{total} checks passed across {', '.join(ENGINES)}")
    for failure in failures:
        print(f"[find-in-page] FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
