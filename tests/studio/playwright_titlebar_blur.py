# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Production chrome/overlay regression scene, with Tauri IPC stubbed.

Runs its own Vite server, or uses SMOKE_URL when supplied.
Set SMOKE_EVIDENCE_DIR to keep screenshots and measured facts.
SMOKE_EXPECT_BLUR=0 captures the same scene against the unmodified base.
No backend, inference, or real native window actions are used.
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args, start_vite, stop_process, wait_for_smoke_page

PORT = int(os.environ.get("SMOKE_PORT", "5491"))
URL = os.environ.get("SMOKE_URL", f"http://127.0.0.1:{PORT}/smoke-titlebar-blur.html")
EXPECTED = os.environ.get("SMOKE_EXPECT_BLUR", "1") == "1"
OUT = Path(
    os.environ.get(
        "SMOKE_EVIDENCE_DIR", str(Path(tempfile.gettempdir()) / "titlebar-blur-evidence")
    )
)
OUT.mkdir(parents = True, exist_ok = True)
INIT = r"""
Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
Object.defineProperty(navigator, 'userAgentData', {get: () => ({platform: 'Windows'})});
window.__windowActions = [];
window.__TAURI_INTERNALS__ = {
  metadata: {currentWindow: {label: 'main'}, currentWebview: {label: 'main'}},
  transformCallback: () => 1, unregisterCallback: () => {},
  invoke: async (cmd, args) => {
    window.__windowActions.push({cmd, args});
    if (cmd.endsWith('is_maximized')) return false;
    if (cmd.endsWith('is_resizable') || cmd.endsWith('is_focused')) return true;
    if (cmd === 'plugin:event|listen') return 1;
    return null;
  }
};
window.__TAURI_EVENT_PLUGIN_INTERNALS__ = {unregisterListener: () => {}};
"""


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(args = chromium_launch_args())
        context = browser.new_context(
            viewport = {"width": 1440, "height": 900}, device_scale_factor = 1
        )
        context.add_init_script(INIT)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda e: (errors.append(str(e)), print("PAGE ERROR:", e)))
        page.route(
            re.compile(r"^http://127\.0\.0\.1:\d+/api/"), lambda route: route.fulfill(json = [])
        )
        print("Loading", URL)
        page.on(
            "response",
            lambda response: errors.append(f"{response.status}: {response.url}")
            if response.status >= 400
            and response.request.resource_type in ("script", "stylesheet", "font")
            else None,
        )
        page.goto(URL)
        titlebar = page.locator('header[aria-label="Window titlebar"]')
        expect(titlebar).to_be_visible(timeout = 60000)
        page.get_by_role("button", name = "Open media", exact = True).click()
        expect(page.get_by_role("dialog")).to_be_visible()
        page.wait_for_timeout(500)
        print("Media open")

        def facts():
            return titlebar.evaluate("""e => {const s = getComputedStyle(e, '::after'); return {
                titlebarZ: getComputedStyle(e).zIndex, content: s.content,
                opacity: s.opacity, blur: s.backdropFilter, pointerEvents: s.pointerEvents,
                backdropZ: s.zIndex,
                controlsZ: getComputedStyle(e.querySelector('[aria-label="Window controls"]')).zIndex,
                overlayZ: getComputedStyle(document.querySelector('[data-slot="dialog-overlay"]')).zIndex
            }}""")

        def blurred(want):
            # Wait for the transition rather than assuming a fixed frame budget.
            # Compositors can report an epsilon near zero on the final frame.
            if not EXPECTED:
                page.wait_for_timeout(180)
                return
            page.wait_for_function(
                """want => {
                    const e = document.querySelector('header[aria-label="Window titlebar"]');
                    const value = Number(getComputedStyle(e, '::after').opacity);
                    return Math.abs(value - (want ? 1 : 0)) < 0.0001;
                }""",
                arg = want,
                timeout = 5000,
            )

        blurred(True)
        media_facts = facts()
        if EXPECTED:
            assert media_facts["blur"] == "blur(2px)", media_facts
            assert media_facts["pointerEvents"] == "none", media_facts
            assert int(media_facts["controlsZ"]) > int(media_facts["backdropZ"]), media_facts
        else:
            assert media_facts["content"] == "none", media_facts
        page.evaluate("document.fonts.ready")
        page.screenshot(path = str(OUT / "media-light.png"))
        page.screenshot(
            path = str(OUT / "titlebar-light.png"),
            clip = {"x": 0, "y": 0, "width": 1440, "height": 150},
        )
        page.evaluate("document.documentElement.classList.add('dark')")
        page.wait_for_timeout(180)
        page.screenshot(path = str(OUT / "media-dark.png"))
        page.evaluate("document.documentElement.classList.remove('dark')")
        page.keyboard.press("Escape")
        blurred(False)
        expect(page.get_by_role("dialog")).to_have_count(0)
        print("Starting modal checks")
        tour_facts = None
        for kind in ["dialog", "alert", "sheet", "tour", "scoped"]:
            print("Checking", kind)
            page.get_by_role("button", name = f"Open {kind}", exact = True).click()
            blurred(kind != "scoped")
            if kind == "dialog":
                page.get_by_role("button", name = "Open nested", exact = True).click()
                blurred(True)
                page.get_by_role("button", name = "Cancel", exact = True).click()
                blurred(True)
            if kind == "alert":
                page.get_by_role("button", name = "Cancel", exact = True).click()
            elif kind == "tour":
                tour_facts = facts()
                if EXPECTED:
                    assert tour_facts["blur"] == "blur(2px)", tour_facts
                    assert tour_facts["opacity"] == "1", tour_facts
                else:
                    assert tour_facts["content"] == "none", tour_facts
                page.screenshot(path = str(OUT / "tour-light.png"))
                page.get_by_role("button", name = "Skip", exact = True).click()
            elif kind in ("dialog", "scoped"):
                page.locator("[data-slot=dialog-close]").click()
            else:
                page.keyboard.press("Escape")
            blurred(False)
        page.get_by_role("button", name = "Open menu", exact = True).click()
        expect(page.get_by_role("menu")).to_be_visible()
        blurred(False)
        page.keyboard.press("Escape")
        commands = []
        for label, command in [
            ("Minimize window", "minimize"),
            ("Maximize window", "toggle_maximize"),
            ("Close window", "close"),
        ]:
            page.get_by_role("button", name = "Open media", exact = True).click()
            blurred(True)
            # Native controls are intentionally pointer-accessible even while Radix hides the background from AT.
            page.locator(f'button[aria-label="{label}"]').click()
            page.wait_for_function(
                "c => window.__windowActions.some(a => a.cmd === 'plugin:window|' + c)", arg = command
            )
            commands.append(command)
            page.keyboard.press("Escape")
            blurred(False)
        page.get_by_role("button", name = "Open media", exact = True).click()
        blurred(True)
        page.mouse.move(600, 20)
        page.mouse.down()
        page.mouse.move(640, 25)
        page.mouse.up()
        page.wait_for_function(
            "() => window.__windowActions.some(a => a.cmd === 'plugin:window|start_dragging')"
        )
        page.keyboard.press("Escape")
        blurred(False)
        page.set_viewport_size({"width": 640, "height": 700})
        page.get_by_role("button", name = "Open media", exact = True).click()
        blurred(True)
        page.screenshot(path = str(OUT / "media-narrow.png"))
        # Web and native macOS chrome do not acquire the custom titlebar effect.
        for platform in ("web", "macOS"):
            other = browser.new_context()
            if platform == "macOS":
                other.add_init_script(INIT.replace("Win32", "MacIntel").replace("Windows", "macOS"))
            web = other.new_page()
            web.route(
                re.compile(r"^http://127\.0\.0\.1:\d+/api/"), lambda route: route.fulfill(json = [])
            )
            web.goto(URL)
            web.get_by_role("button", name = "Open media", exact = True).click()
            expect(web.get_by_role("dialog")).to_be_visible()
            expect(web.locator('header[aria-label="Window titlebar"]')).to_have_count(0)
            other.close()
        assert not errors, errors
        (OUT / "facts.json").write_text(
            json.dumps(
                {
                    "url": URL,
                    "expected_blur": EXPECTED,
                    "media": media_facts,
                    "tour": tour_facts,
                    "commands": commands + ["start_dragging"],
                    "checks": [
                        "media",
                        "dialog",
                        "alert",
                        "sheet",
                        "tour",
                        "scoped excluded",
                        "menu excluded",
                        "nested",
                        "close cleanup",
                        "dark",
                        "narrow",
                        "web",
                        "macOS",
                    ],
                    "page_errors": errors,
                },
                indent = 2,
            )
        )
        browser.close()
        print(json.dumps({"passed": True, "facts": media_facts, "output": str(OUT)}))


if __name__ == "__main__":
    server = None if os.environ.get("SMOKE_URL") else start_vite(PORT)
    try:
        if server is not None:
            # start_vite returns as soon as npm is spawned; navigating before vite listens is a
            # connection refused (#12475's run), so wait for the page that names this scene's entry.
            wait_for_smoke_page(URL, "smoke-titlebar-blur-main.tsx", proc = server)
        run()
    finally:
        if server is not None:
            stop_process(server)
