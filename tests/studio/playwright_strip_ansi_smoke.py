# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser smoke for #7962: production tool-output panes strip ANSI before render."""

from __future__ import annotations

import os

import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

# 8000 collides with whatever else is on a shared box; sit by chat (5193) and research (5183).
PORT = int(os.environ.get("SMOKE_PORT", "5203"))
BASE = os.environ.get("SMOKE_BASE_URL", f"http://127.0.0.1:{PORT}")
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-ansi-smoke"))
SECTIONS = (
    "tool-result-output",
    "tool-fallback-result",
    "tool-live-output",
    "code-execution-result",
    "reconciled-terminal-result",
)
ESC = "\u001b"


def info(msg: str) -> None:
    print(f"[ansi-smoke] {msg}", flush = True)


def main() -> None:
    ART.mkdir(parents = True, exist_ok = True)
    info(f"starting vite dev server on port {PORT}")
    vite = start_vite(PORT)
    try:
        wait_for_smoke_page(f"{BASE}/smoke-ansi.html", "smoke-ansi-main.tsx", info = info)

        with sync_playwright() as playwright:
            browser_name = os.environ.get("PW_BROWSER", "chromium").lower()
            if browser_name not in {"chromium", "firefox", "webkit"}:
                raise ValueError(f"unsupported PW_BROWSER: {browser_name}")
            browser_type = getattr(playwright, browser_name)
            launch_args = chromium_launch_args() if browser_name == "chromium" else []
            browser = browser_type.launch(headless = True, args = launch_args)
            page = browser.new_page()
            page.goto(f"{BASE}/smoke-ansi.html", wait_until = "networkidle")
            page.screenshot(path = str(ART / "smoke-ansi.png"), full_page = True)

            for section in SECTIONS:
                pane = page.locator(f'section[data-smoke="{section}"] pre').first
                expect(pane).to_be_visible(timeout = 15_000)
                text = pane.inner_text()
                info(f"{section} text: {text!r}")
                assert text == "file.txt\nerror", f"{section} rendered unexpected text: {text!r}"
                assert ESC not in text, f"{section} still contains ESC"
                assert "[32m" not in text, f"{section} still shows SGR garbage"

            info("all production panes rendered clean text (no ANSI escapes)")
            browser.close()
    finally:
        stop_process(vite)


if __name__ == "__main__":
    main()
