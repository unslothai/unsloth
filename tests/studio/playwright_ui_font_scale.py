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
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import wait_for_health  # noqa: E402

BASE = os.environ["BASE_URL"]
PW = os.environ["STUDIO_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_fontscale"))
ART.mkdir(parents = True, exist_ok = True)

SIZES = (12, 20)
DEFAULT = 16


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
    page.keyboard.press("Control+,")
    page.wait_for_timeout(700)
    if page.get_by_role("dialog").count() == 0:
        page.keyboard.press("Meta+,")
        page.wait_for_timeout(700)
    if page.get_by_role("dialog").count() == 0:
        fail("settings dialog did not open")
    page.get_by_role("dialog").get_by_role("button").filter(has_text = "Appearance").first.click()
    page.wait_for_timeout(600)


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
        page.get_by_role("dialog").get_by_role("button").filter(has_text = "Voice").first.click()
        page.wait_for_timeout(600)
        page.set_viewport_size({"width": 1440, "height": 480})
        page.locator("[aria-label='Dictation language']").click()
        page.wait_for_timeout(700)
        state = page.evaluate(
            """
            () => {
              const vp = document.querySelector("[data-radix-select-viewport]");
              return vp
                ? { scrollable: vp.scrollHeight > vp.clientHeight, top: vp.scrollTop }
                : null;
            }
            """
        )
        if not state or not state["scrollable"]:
            fail(f"select viewport not scrollable: {state}")
        for _ in range(6):
            page.keyboard.press("ArrowDown")
            page.wait_for_timeout(100)
        kb_top = page.evaluate(
            "() => document.querySelector('[data-radix-select-viewport]').scrollTop"
        )
        if not kb_top > 0:
            fail(f"keyboard did not scroll the select viewport: {kb_top}")
        vp_box = page.locator("[data-radix-select-viewport]").bounding_box()
        page.mouse.move(vp_box["x"] + vp_box["width"] / 2, vp_box["y"] + 40)
        page.mouse.wheel(0, -400)
        page.wait_for_timeout(300)
        wheel_top = page.evaluate(
            "() => document.querySelector('[data-radix-select-viewport]').scrollTop"
        )
        if not wheel_top < kb_top:
            fail(f"wheel did not scroll the select viewport: {kb_top} -> {wheel_top}")
        page.keyboard.press("Escape")
        page.set_viewport_size({"width": 1440, "height": 900})
        page.wait_for_timeout(400)

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
