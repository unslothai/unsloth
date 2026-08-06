# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser smoke for #7962: tool output panes strip ANSI before render."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args  # noqa: E402

FRONTEND = Path(__file__).resolve().parents[2] / "studio" / "frontend"
BASE = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:8000")
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-ansi-smoke"))
ESC = "\u001b"


def info(msg: str) -> None:
    print(f"[ansi-smoke] {msg}", flush = True)


def wait_for_vite(url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/smoke-ansi.html", timeout = 2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"vite dev server did not become ready at {url}")


def main() -> None:
    ART.mkdir(parents = True, exist_ok = True)
    info(f"starting vite dev server in {FRONTEND}")
    vite = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "8000", "--strictPort"],
        cwd = FRONTEND,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )
    try:
        wait_for_vite(BASE)
        info(f"vite ready at {BASE}")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless = True,
                args = chromium_launch_args(),
            )
            page = browser.new_page()
            page.goto(f"{BASE}/smoke-ansi.html", wait_until = "networkidle")
            page.screenshot(path = str(ART / "smoke-ansi.png"), full_page = True)

            for section in ("tool-result-output", "tool-fallback-result"):
                pane = page.locator(f'section[data-smoke="{section}"] pre').first
                expect(pane).to_be_visible(timeout = 10_000)
                text = pane.inner_text()
                info(f"{section} text: {text!r}")
                assert ESC not in text, f"{section} still contains ESC"
                assert "file.txt" in text, f"{section} missing visible filename"
                assert "error" in text, f"{section} missing visible error word"
                assert "[32m" not in text, f"{section} still shows SGR garbage"

            info("all panes rendered clean text (no ANSI escapes)")
            browser.close()
    finally:
        vite.terminate()
        try:
            vite.wait(timeout = 10)
        except subprocess.TimeoutExpired:
            vite.kill()


if __name__ == "__main__":
    main()
