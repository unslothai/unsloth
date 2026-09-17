# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UI font size scaling regression (Settings > Appearance).

Drives the real appearance controls and asserts the typography-scale
contract: text and line heights scale by size/16, the root font size and
layout geometry never move, an explicit Code font size stays fixed, and an
overflowing Radix select scrolls its viewport by keyboard and wheel.

Runs against an already-booted, already-bootstrapped Unsloth:
    BASE_URL=http://127.0.0.1:18894 STUDIO_PW=... python tests/studio/playwright_ui_font_scale.py
"""

import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import wait_for_health  # noqa: E402

BASE = os.environ["BASE_URL"]
PW = os.environ["STUDIO_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_fontscale"))
ART.mkdir(parents = True, exist_ok = True)

# Read the range from the store instead of restating it: the default is the one size at which data-ui-font-size is
# dropped, and it has already moved once (16 -> 15), which is exactly what a pinned copy here fails on.
_STORE = (
    Path(__file__).resolve().parents[2]
    / "studio/frontend/src/features/settings/stores/appearance-custom-store.ts"
).read_text(encoding = "utf-8")
_RANGE = re.search(
    r"UI_FONT_SIZE_RANGE\s*=\s*\{\s*min:\s*(\d+),\s*max:\s*(\d+),\s*default:\s*(\d+)",
    _STORE,
)
if _RANGE is None:
    raise AssertionError("[font-scale] FAIL: no UI_FONT_SIZE_RANGE in appearance-custom-store.ts")
SIZES = (int(_RANGE.group(1)), int(_RANGE.group(2)))
DEFAULT = int(_RANGE.group(3))
# The base the authored rem typography is written against; --ui-font-scale is the preference divided by it.
_BASE = re.search(r"UI_FONT_SIZE_CSS_BASE\s*=\s*(\d+)", _STORE)
if _BASE is None:
    raise AssertionError(
        "[font-scale] FAIL: no UI_FONT_SIZE_CSS_BASE in appearance-custom-store.ts"
    )
CSS_BASE = int(_BASE.group(1))


def settled_scroll_top(
    page,
    quiet_ms = 200,
    timeout_ms = 5_000,
):
    """The select viewport's scrollTop once it has stopped moving.

    Radix scrolls the highlighted item into view off the back of the keypress, so
    a scrollTop read straight after `keyboard.press` is a mid-scroll sample, not
    where the viewport ends up. Poll until it holds the same value for `quiet_ms`.
    Falls back to the last value seen rather than raising: this only establishes
    the floor for the wheel check, and that check reports its own failure.
    """
    last = page.evaluate(SCROLL_TOP_JS)
    quiet_since = time.monotonic()
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        page.wait_for_timeout(50)
        now = page.evaluate(SCROLL_TOP_JS)
        if now != last:
            last = now
            quiet_since = time.monotonic()
        elif (time.monotonic() - quiet_since) * 1000 >= quiet_ms:
            return now
    return last


def step(s):
    print(f"[font-scale] STEP {s}", flush = True)


def fail(m):
    raise AssertionError(f"[font-scale] FAIL: {m}")


def near(
    a,
    b,
    tol = 0.35,
):
    return a is not None and b is not None and abs(a - b) <= tol


_VP = 'document.querySelector("[data-radix-select-viewport]")'
SCROLL_TOP_JS = f"() => {_VP}.scrollTop"
SCROLLABLE_JS = f"() => {{ const vp = {_VP}; return !!vp && vp.scrollHeight > vp.clientHeight; }}"
VIEWPORT_STATE_JS = f"""
() => {{
  const vp = {_VP};
  return vp
    ? {{ scrollHeight: vp.scrollHeight, clientHeight: vp.clientHeight, top: vp.scrollTop }}
    : null;
}}
"""

MEASURE_JS = """
() => {
  const fs = (el) => (el ? parseFloat(getComputedStyle(el).fontSize) : null);
  const lh = (el) => (el ? parseFloat(getComputedStyle(el).lineHeight) : null);
  const byText = (txt) =>
    [...document.querySelectorAll("span, h2, label, p")].find(
      (e) => e.textContent.trim() === txt,
    );
  const nav = byText("New chat");
  const sidebar =
    document.querySelector("[data-slot='sidebar-container']") ??
    document.querySelector("aside") ??
    document.querySelector("nav");
  return {
    root: parseFloat(getComputedStyle(document.documentElement).fontSize),
    uiAttr: document.documentElement.getAttribute("data-ui-font-size"),
    navFont: fs(nav),
    navLine: lh(nav),
    sidebarW: sidebar ? sidebar.getBoundingClientRect().width : null,
  };
}
"""


def measure(page):
    return page.evaluate(MEASURE_JS)


def set_input(page, label, value):
    field = page.locator(f"input[aria-label='{label}']")
    field.scroll_into_view_if_needed()
    field.fill(str(value))
    page.keyboard.press("Enter")
    page.wait_for_timeout(600)


def open_appearance(page):
    # The shortcut can fire before the app has wired its key handler, so press each chord once behind a fixed sleep and
    # a slow boot loses the dialog. Alternate them on a bounded retry, waiting on the dialog itself.
    dialog = page.get_by_role("dialog")
    for attempt in range(10):
        page.keyboard.press("Meta+," if attempt % 2 else "Control+,")
        try:
            dialog.first.wait_for(state = "visible", timeout = 2_000)
            break
        except PWTimeout:
            continue
    if dialog.count() == 0:
        fail("settings dialog did not open after 10 attempts")
    dialog.get_by_role("button").filter(has_text = "Appearance").first.click()
    # Wait for the control the caller is about to drive, not a fixed interval.
    page.locator("input[aria-label='UI font size']").wait_for(state = "visible", timeout = 15_000)


def main():
    wait_for_health(BASE)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport = {"width": 1440, "height": 900})
        page.goto(BASE, wait_until = "networkidle")
        pw_field = page.locator("input[type='password']")
        if pw_field.count():
            pw_field.first.fill(PW)
            page.keyboard.press("Enter")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(1500)

        step("baseline at the default size")
        open_appearance(page)
        set_input(page, "UI font size", DEFAULT)
        base = measure(page)
        if base["root"] != 16:
            fail(f"root font size not 16 at default: {base['root']}")
        if base["navFont"] is None or base["sidebarW"] is None:
            fail(f"baseline samples missing: {base}")

        for size in SIZES:
            step(f"UI font size {size}")
            set_input(page, "UI font size", size)
            m = measure(page)
            ratio = size / DEFAULT
            if m["root"] != 16:
                fail(f"root font size moved at {size}: {m['root']}")
            if m["uiAttr"] != str(size):
                fail(f"data-ui-font-size wrong at {size}: {m['uiAttr']}")
            if not near(m["navFont"], base["navFont"] * ratio):
                fail(f"nav font at {size}: {base['navFont']} -> {m['navFont']}")
            if not near(m["navLine"], base["navLine"] * ratio):
                fail(f"nav line-height at {size}: {base['navLine']} -> {m['navLine']}")
            if not near(m["sidebarW"], base["sidebarW"], 0.75):
                fail(f"sidebar width moved at {size}: {base['sidebarW']} -> {m['sidebarW']}")
            page.screenshot(path = str(ART / f"scale-{size}.png"))

        step("explicit Code font size stays fixed under UI 20")
        set_input(page, "Code font size", 13)
        res = page.evaluate(
            """
            () => {
              const pre = document.createElement("pre");
              pre.textContent = "sample";
              document.body.appendChild(pre);
              const size = getComputedStyle(pre).fontSize;
              pre.remove();
              return size;
            }
            """
        )
        if res != "13px":
            fail(f"explicit code font size scaled: {res}")
        code_field = page.locator("input[aria-label='Code font size']")
        code_field.fill("")
        page.keyboard.press("Enter")
        page.wait_for_timeout(400)

        step("overflowing select scrolls its Radix viewport")
        voice = page.get_by_role("dialog").get_by_role("button").filter(has_text = "Voice").first
        voice.click()
        page.set_viewport_size({"width": 1440, "height": 480})
        trigger = page.locator("[aria-label='Dictation language']")
        trigger.wait_for(state = "visible")
        trigger.click()

        viewport = page.locator("[data-radix-select-viewport]")
        viewport.wait_for(state = "visible")
        # Wait for the overflow itself rather than a fixed sleep: the list is populated asynchronously, so measuring
        # too early reads it as short.
        try:
            page.wait_for_function(SCROLLABLE_JS, timeout = 10_000)
        except PWTimeout:
            fail(f"select viewport not scrollable: {page.evaluate(VIEWPORT_STATE_JS)}")

        # Radix moves focus into the listbox after the content opens, so a fixed
        # burst of presses can land on the trigger and scroll nothing. Press until
        # it moves instead; a real regression still fails, just after more tries.
        kb_top = 0
        for _ in range(40):
            page.keyboard.press("ArrowDown")
            kb_top = page.evaluate(SCROLL_TOP_JS)
            if kb_top > 0:
                break
            page.wait_for_timeout(50)
        if not kb_top > 0:
            fail(f"keyboard did not scroll the select viewport after 40 presses: {kb_top}")

        # That read lands mid-scroll and comes in low (24-35px on the ubuntu CI image), which is neither the floor the
        # wheel has to beat nor a moment a wheel event survives. Let the scroll finish and re-read instead of racing it.
        kb_top = settled_scroll_top(page)
        # At 0 the comparison below is unsatisfiable, so the wheel would always fail.
        if not kb_top > 0:
            fail(f"select viewport returned to the top once the keyboard scroll settled: {kb_top}")

        vp_box = viewport.bounding_box()
        # Keep the pointer inside the viewport: a fixed 40px offset lands outside a shorter box and the wheel then goes
        # to whatever is underneath.
        page.mouse.move(
            vp_box["x"] + vp_box["width"] / 2,
            vp_box["y"] + min(40, vp_box["height"] / 2),
        )
        # A single wheel event can be dropped, so retry a bounded number of times. A
        # viewport that truly refuses the wheel never moves and still fails.
        wheel_top = kb_top
        for _ in range(5):
            page.mouse.wheel(0, -400)
            try:
                page.wait_for_function(
                    "top => document.querySelector('[data-radix-select-viewport]').scrollTop < top",
                    arg = kb_top,
                    timeout = 2_000,
                )
                break
            except PWTimeout:
                wheel_top = page.evaluate(SCROLL_TOP_JS)
        else:
            fail(f"wheel did not scroll the select viewport: {kb_top} -> {wheel_top}")
        page.keyboard.press("Escape")
        page.set_viewport_size({"width": 1440, "height": 900})
        page.wait_for_timeout(400)

        step("cn keeps text-ui-* next to color classes (hub tabs)")
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        page.goto(f"{BASE}/hub", wait_until = "domcontentloaded")
        page.wait_for_timeout(2000)
        open_appearance(page)
        small = SIZES[0]
        set_input(page, "UI font size", small)
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        tab = page.get_by_role("radio").filter(has_text = "Discover").first
        tab.wait_for(state = "visible", timeout = 15000)
        tab_font = tab.evaluate("el => parseFloat(getComputedStyle(el).fontSize)")
        # text-ui-12p5 at the smallest scale; the unscaled 12.5px means twMerge dropped the token.
        if not near(tab_font, 12.5 * small / CSS_BASE):
            fail(f"hub tab font did not scale (twMerge drop?): {tab_font}")
        icon_w = page.evaluate(
            "() => { const el = document.querySelector('.size-icon');"
            " return el ? parseFloat(getComputedStyle(el).width) : null; }"
        )
        # Standard icons render at the UI font size itself below the CSS base, so the smallest setting gives glyphs of
        # exactly that many px.
        if not near(icon_w, small):
            fail(f"size-icon did not match the UI font size below {CSS_BASE}: {icon_w}")
        page.goto(BASE, wait_until = "domcontentloaded")
        page.wait_for_timeout(1500)
        open_appearance(page)

        step("default restores exactly")
        page.get_by_role("dialog").get_by_role("button").filter(has_text = "Appearance").first.click()
        page.wait_for_timeout(500)
        set_input(page, "UI font size", DEFAULT)
        final = measure(page)
        for key in ("root", "navFont", "navLine", "sidebarW"):
            if not near(final[key], base[key], 0.35):
                fail(f"default drifted for {key}: {base[key]} -> {final[key]}")
        if final["uiAttr"] is not None:
            fail(f"data-ui-font-size present at default: {final['uiAttr']}")

        page.screenshot(path = str(ART / "restored-default.png"))
        browser.close()
    print("[font-scale] PASS", flush = True)


if __name__ == "__main__":
    main()
