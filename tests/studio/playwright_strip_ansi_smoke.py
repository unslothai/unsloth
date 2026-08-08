# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser smoke for #7962: production tool-output panes strip ANSI before render."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
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
SECTIONS = (
    "tool-result-output",
    "tool-fallback-result",
    "tool-live-output",
    "code-execution-result",
)
ESC = "\u001b"


def info(msg: str) -> None:
    print(f"[ansi-smoke] {msg}", flush=True)


def wait_for_vite(timeout_s: float = 120.0) -> None:
    url = f"{BASE}/smoke-ansi.html"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"vite dev server did not become ready at {url}")


def drain_process_output(proc: subprocess.Popen[str]) -> None:
    if proc.stdout is not None:
        for _ in proc.stdout:
            pass


def start_vite() -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "8000", "--strictPort"],
        cwd=FRONTEND,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    threading.Thread(target=drain_process_output, args=(proc,), daemon=True).start()
    return proc


def stop_process(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    info(f"starting vite dev server in {FRONTEND}")
    vite = start_vite()
    try:
        wait_for_vite()
        info(f"vite ready at {BASE}")

        with sync_playwright() as playwright:
            browser_name = os.environ.get("PW_BROWSER", "chromium").lower()
            if browser_name not in {"chromium", "firefox", "webkit"}:
                raise ValueError(f"unsupported PW_BROWSER: {browser_name}")
            browser_type = getattr(playwright, browser_name)
            launch_args = chromium_launch_args() if browser_name == "chromium" else []
            browser = browser_type.launch(headless=True, args=launch_args)
            page = browser.new_page()
            page.goto(f"{BASE}/smoke-ansi.html", wait_until="networkidle")
            page.screenshot(path=str(ART / "smoke-ansi.png"), full_page=True)

            for section in SECTIONS:
                pane = page.locator(f'section[data-smoke="{section}"] pre').first
                expect(pane).to_be_visible(timeout=15_000)
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
