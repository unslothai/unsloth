# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise find-in-page in real browsers and degraded engine modes.

The Node suite covers pure indexing. This harness covers browser-only chords, geometry,
highlights, fallback selection, and visibility behavior.

    SMOKE_ENGINES=chromium,firefox,webkit python3 tests/studio/playwright_find_in_page.py

The modes emulate missing highlight support, legacy visibility options, and no visibility API."""

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

ENTRY_FAIL = os.environ.get("SMOKE_ENTRY_FAIL", "")

ENTRY_DELAY_MS = int(os.environ.get("SMOKE_ENTRY_DELAY_MS", "0"))
ENTRY_SCREENSHOT = os.environ.get("SMOKE_ENTRY_SCREENSHOT", "")
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
    return page.evaluate("() => window.__findSmoke.state()")


def open_bar(page, mod: str) -> None:
    page.keyboard.press(f"{mod}+f")
    # The production bar crosses a lazy boundary so the first open can include
    # one dev-server transform. Assert that it resolves, not that Vite is warm.
    page.wait_for_function(
        "() => window.__findSmoke.state().open",
        timeout = 10000,
    )


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

    # Re-pressing the chord keeps the search open and returns focus to its field.
    page.evaluate("() => document.activeElement?.blur()")
    check(
        engine,
        mode,
        "the field can lose focus while the bar stays open",
        state(page).get("focused") is False,
    )
    page.keyboard.press(f"{mod}+f")
    page.wait_for_timeout(200)
    check(
        engine,
        mode,
        "the chord re-focuses the field instead of closing",
        state(page).get("open") is True and state(page).get("focused") is True,
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
    page.keyboard.press(f"{mod}+f")
    page.wait_for_timeout(250)
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
    """Verify display-none, hidden visibility, and hidden contents text are excluded."""
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
    """Verify a match below a `content-visibility: auto` placeholder is revealed."""
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
    """Verify composed and decomposed spellings share a correctly mapped range."""
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
    # Only the registry path exposes a range; fallback selection is intentionally released so the
    # field remains usable.
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


def check_typing_burst(page, engine: str, mode: str, mod: str) -> None:
    """A word typed quickly searches once, without stalling each following key."""
    open_bar(page, mod)
    field = page.locator('[role="search"] input')
    field.fill("")
    page.wait_for_timeout(150)
    page.evaluate(
        """() => {
      window.__findSmokePaints = 0;
      if (CSS.highlights) {
        const registry = CSS.highlights;
        const set = registry.set.bind(registry);
        registry.set = (...args) => {
          window.__findSmokePaints += 1;
          return set(...args);
        };
      } else {
        const addRange = Selection.prototype.addRange;
        Selection.prototype.addRange = function (...args) {
          window.__findSmokePaints += 1;
          return addRange.apply(this, args);
        };
      }
    }"""
    )
    page.keyboard.type("doing", delay = 10)
    immediate = field.input_value()
    page.wait_for_timeout(400)
    paints = page.evaluate("() => window.__findSmokePaints")
    shown = counter(page)
    check(
        engine,
        mode,
        "a typing burst leaves every character responsive",
        immediate == "doing",
        f"value={immediate!r}",
    )
    check(
        engine,
        mode,
        "a typing burst coalesces to one search paint",
        paints <= 2 and bool(shown),
        f"paint operations={paints}, counter={shown!r}",
    )
    close_bar(page)


def check_reverted_query_repaints(page, engine: str, mode: str, mod: str) -> None:
    """Restoring the settled query inside the debounce window must restore its paint."""
    if not page.evaluate("() => typeof CSS !== 'undefined' && !!CSS.highlights"):
        return
    open_bar(page, mod)
    field = page.locator('[role="search"] input')
    field.fill("unsloth")
    page.wait_for_function(
        "() => !!window.__findSmoke.counter()",
        timeout = 10000,
    )
    page.wait_for_timeout(150)
    before = page.evaluate(
        """() => CSS.highlights.has('unsloth-find') &&
          CSS.highlights.has('unsloth-find-active')"""
    )
    field.press("End")
    field.type("x")
    page.wait_for_timeout(20)
    field.press("Backspace")
    page.wait_for_timeout(250)
    after = page.evaluate(
        """() => ({
          painted: CSS.highlights.has('unsloth-find') &&
            CSS.highlights.has('unsloth-find-active'),
          counter: window.__findSmoke.counter(),
        })"""
    )
    check(
        engine,
        mode,
        "returning to the settled query repaints its unchanged matches",
        before is True and field.input_value() == "unsloth" and after["painted"] is True,
        f"painted before={before}, after={after}, value={field.input_value()!r}",
    )
    close_bar(page)


def check_pending_query_stays_unpainted(page, engine: str, mode: str, mod: str) -> None:
    """Keeps old highlights from repainting during a typing burst."""
    if not page.evaluate("() => typeof CSS !== 'undefined' && !!CSS.highlights"):
        return
    open_bar(page, mod)
    field = page.locator('[role="search"] input')
    field.fill("unsloth")
    page.wait_for_function("() => !!window.__findSmoke.counter()", timeout = 10000)
    page.wait_for_timeout(150)
    field.press("End")
    field.type("x")
    page.evaluate(
        """() => {
      document.getElementById('pending-repaint-probe')?.remove();
      const row = document.createElement('p');
      row.id = 'pending-repaint-probe';
      row.textContent = 'mutation while the query remains unsettled';
      document.querySelector('[data-find-scope]').appendChild(row);
    }"""
    )
    field.type("abcdefghi", delay = 55)
    pending = {
        "value": field.input_value(),
        "counter": counter(page),
        "painted": page.evaluate(
            """() => CSS.highlights.has('unsloth-find') ||
              CSS.highlights.has('unsloth-find-active')"""
        ),
    }
    check(
        engine,
        mode,
        "an asynchronous reindex stays unpainted while the query is pending",
        pending["value"] != "unsloth"
        and pending["counter"] in (None, "")
        and not pending["painted"],
        f"pending state={pending}",
    )
    page.wait_for_timeout(150)
    close_bar(page)
    page.evaluate("() => document.getElementById('pending-repaint-probe')?.remove()")


def check_katex_mutation_reindexes(page, engine: str, mode: str, mod: str) -> None:
    """A text mutation in KaTeX's painted aria-hidden tree invalidates the index."""
    page.evaluate(
        """() => {
      const scope = document.querySelector('[data-find-scope]') ?? document.body;
      document.getElementById('probe-katex-mutation')?.remove();
      const host = document.createElement('span');
      host.id = 'probe-katex-mutation';
      host.className = 'katex';
      host.innerHTML = '<span class="katex-mathml">ignored mirror</span>' +
        '<span class="katex-html" aria-hidden="true">mutablekatex</span>';
      scope.appendChild(host);
    }"""
    )
    page.wait_for_timeout(500)
    open_bar(page, mod)
    page.locator('[role="search"] input').fill("mutablekatex")
    page.wait_for_function(
        "() => window.__findSmoke.counter() === '1/1'",
        timeout = 10000,
    )
    page.evaluate(
        "() => { document.querySelector('#probe-katex-mutation .katex-html').textContent = 'changedkatex'; }"
    )
    page.wait_for_timeout(900)
    after = counter(page)
    check(
        engine,
        mode,
        "a KaTeX painted-text mutation refreshes the count",
        after in (None, "", "0/0"),
        f"counter={after!r} (1/1 means aria-hidden mutation filtering stayed stale)",
    )
    close_bar(page)
    page.evaluate("() => document.getElementById('probe-katex-mutation')?.remove()")


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
        check_typing_burst,
        check_reverted_query_repaints,
        check_pending_query_stays_unpainted,
        check_katex_mutation_reindexes,
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


def run_entry_chunk_failure(browser, engine: str) -> None:
    """A first-use find chunk failure degrades locally instead of unmounting the shell."""
    context = browser.new_context(user_agent = PLATFORMS["Linux"][1])

    def block_find_entry(route):
        if "find-bar-loader" in route.request.url:
            return route.abort("failed")
        return route.fallback()

    context.route("**/*", block_find_entry)
    page = new_page(context)
    page.keyboard.press("Control+f")
    failure = page.locator('[data-testid="find-in-page-load-failure"]')
    failure.wait_for(state = "visible", timeout = 15000)
    root_text = page.locator("#root").inner_text()
    mode = "entry-chunk-failure"
    check(engine, mode, "the local failure notice is visible", failure.count() == 1)
    check(
        engine,
        mode,
        "the failure offers reload recovery",
        failure.get_by_role("button").count() == 1,
    )
    check(
        engine,
        mode,
        "the conversation shell survives",
        "Message 1" in root_text and "Message 40" in root_text,
        f"root text length={len(root_text)}",
    )
    check(
        engine,
        mode,
        "the failed bar does not leave a search landmark",
        page.locator('[role="search"]').count() == 0,
    )
    if ENTRY_SCREENSHOT:
        Path(ENTRY_SCREENSHOT.format(engine = engine)).parent.mkdir(parents = True, exist_ok = True)
        page.screenshot(path = ENTRY_SCREENSHOT.format(engine = engine), full_page = False)
    context.close()


def run_entry_chunk_delay(browser, engine: str) -> None:
    """Keeps query, focus, selection, and commands through a delayed entry handoff."""
    context = browser.new_context(user_agent = PLATFORMS["Linux"][1])
    page = new_page(context)
    composer = page.locator('textarea[placeholder="Message"]')
    composer.focus()
    started = time.monotonic()
    page.keyboard.press("Control+f")
    loading = page.get_by_test_id("find-in-page-loading")
    loading.wait_for(state = "visible", timeout = 5000)
    field = loading.locator("input")
    page.keyboard.type("unsloth", delay = 15)
    field.press("End")
    for _ in range(3):
        field.press("Shift+ArrowLeft")

    field.press("Enter")
    loading_state = field.evaluate(
        """input => ({
          value: input.value,
          start: input.selectionStart,
          end: input.selectionEnd,
          direction: input.selectionDirection,
        })"""
    )
    composer_value = composer.input_value()
    loading.wait_for(
        state = "detached",
        timeout = max(15000, ENTRY_DELAY_MS + 10000),
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    loaded = page.locator('[role="search"] input')
    loaded_state = loaded.evaluate(
        """input => ({
          value: input.value,
          start: input.selectionStart,
          end: input.selectionEnd,
          direction: input.selectionDirection,
        })"""
    )
    mode = "entry-chunk-delay"
    check(
        engine,
        mode,
        "the controlled loading shell receives early typing",
        loading_state["value"] == "unsloth" and composer_value == "",
        f"loading={loading_state}, composer={composer_value!r}",
    )
    check(
        engine,
        mode,
        "the configured delay holds the real entry chunk",
        elapsed_ms >= ENTRY_DELAY_MS * 0.8,
        f"delay={ENTRY_DELAY_MS}ms, observed={elapsed_ms:.1f}ms",
    )
    check(
        engine,
        mode,
        "the loaded bar retains the query and focus",
        loaded_state["value"] == "unsloth" and state(page).get("focused") is True,
        f"loaded={loaded_state}, state={state(page)}",
    )
    check(
        engine,
        mode,
        "the loaded bar retains the loading input selection",
        loaded_state == loading_state,
        f"loading={loading_state}, loaded={loaded_state}",
    )
    check(
        engine,
        mode,
        "Enter queued by the loading shell advances after handoff",
        "2/28" in page.locator('[role="search"]').inner_text(),
        page.locator('[role="search"]').inner_text(),
    )
    context.close()

    context = browser.new_context(user_agent = PLATFORMS["Linux"][1])
    page = new_page(context)
    page.locator('textarea[placeholder="Message"]').focus()
    page.keyboard.press("Control+f")
    loading = page.get_by_test_id("find-in-page-loading")
    loading.wait_for(state = "visible", timeout = 5000)
    field = loading.locator("input")
    field.fill("definitely-no-such-match")
    field.press("Enter")
    loading.wait_for(
        state = "detached",
        timeout = max(15000, ENTRY_DELAY_MS + 10000),
    )
    loaded = page.locator('[role="search"] input')
    loaded.fill("unsloth")
    page.wait_for_timeout(500)
    check(
        engine,
        mode,
        "a queued step for an empty result does not advance a later query",
        "1/28" in page.locator('[role="search"]').inner_text(),
        page.locator('[role="search"]').inner_text(),
    )
    context.close()

    context = browser.new_context(user_agent = PLATFORMS["Linux"][1])
    page = new_page(context)
    page.locator('textarea[placeholder="Message"]').focus()
    page.keyboard.press("Control+f")
    loading = page.get_by_test_id("find-in-page-loading")
    loading.wait_for(state = "visible", timeout = 5000)
    loading.locator("input").dispatch_event(
        "keydown",
        {
            "key": "Escape",
            "code": "Escape",
            "keyCode": 229,
            "isComposing": True,
            "bubbles": True,
            "cancelable": True,
        },
    )
    page.wait_for_timeout(100)
    check(
        engine,
        mode,
        "a composing Escape leaves the loading find session open",
        loading.count() == 1,
    )
    context.close()

    context = browser.new_context(user_agent = PLATFORMS["Linux"][1])
    page = new_page(context)
    page.locator('textarea[placeholder="Message"]').focus()
    page.keyboard.press("Control+f")
    loading = page.get_by_test_id("find-in-page-loading")
    loading.wait_for(state = "visible", timeout = 5000)
    page.locator("[data-find-scope]").click(position = {"x": 20, "y": 200})
    page.keyboard.press("Escape")
    page.wait_for_timeout(100)
    check(
        engine,
        mode,
        "Escape closes the loading find session after focus leaves it",
        page.locator('[role="search"]').count() == 0,
    )
    context.close()


def run_engine(pw, engine: str) -> None:
    launch = {"args": chromium_launch_args()} if engine == "chromium" else {}
    browser = getattr(pw, engine).launch(**launch)

    if ENTRY_FAIL:
        try:
            run_entry_chunk_failure(browser, engine)
        finally:
            browser.close()
        return

    if ENTRY_DELAY_MS:
        try:
            run_entry_chunk_delay(browser, engine)
        finally:
            browser.close()
        return
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
    if ENTRY_DELAY_MS:
        os.environ["SMOKE_MODULE_DELAY_MATCH"] = "find-bar-loader"
        os.environ["SMOKE_MODULE_DELAY_MS"] = str(ENTRY_DELAY_MS)

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
