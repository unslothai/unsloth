# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser smoke for #7962: production tool-output panes strip ANSI before render."""

from __future__ import annotations

import os

import subprocess
import sys
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    dump_diagnostics,
    echo_browser_errors,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

# 8000 collides with whatever else is on a shared box; sit by chat (5193) and research (5183).
PORT = int(os.environ.get("SMOKE_PORT", "5203"))
# Unset: start and stop our own server. Set: drive that one and leave it running.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
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


def dump(page: Page, vite: subprocess.Popen[str] | None) -> None:
    """Write down what the page actually was, since CI keeps no live browser.

    `dump_diagnostics` records the browser side (screenshot, URL, body excerpt). The
    dev server's own output is the other half: a transform error or a forced reload
    is reported there and nowhere else.
    """
    dump_diagnostics(page, ART, "smoke-ansi-failure", info = info)
    if vite is not None:
        info("vite tail:")
        # Snapshot first:
        # Snapshot first: the drain thread is still appending, and printing releases the GIL, so lazy iteration raises
        for line in list(getattr(vite, "vite_tail", [])) or ["(no output)"]:
            info(f"  {line.rstrip()}")
    info(f"artifacts in {ART}")


def main() -> None:
    ART.mkdir(parents = True, exist_ok = True)
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
    vite = start_vite(PORT) if OWNS_SERVER else None
    try:
        wait_for_smoke_page(f"{BASE}/smoke-ansi.html", "smoke-ansi-main.tsx", proc = vite, info = info)

        with sync_playwright() as playwright:
            browser_name = os.environ.get("PW_BROWSER", "chromium").lower()
            if browser_name not in {"chromium", "firefox", "webkit"}:
                raise ValueError(f"unsupported PW_BROWSER: {browser_name}")
            browser_type = getattr(playwright, browser_name)
            launch_args = chromium_launch_args() if browser_name == "chromium" else []
            browser = browser_type.launch(headless = True, args = launch_args)
            page = browser.new_page()
            echo_browser_errors(page, info)
            try:
                page.goto(f"{BASE}/smoke-ansi.html", wait_until = "networkidle")
                page.screenshot(path = str(ART / "smoke-ansi.png"), full_page = True)

                for section in SECTIONS:
                    pane = page.locator(f'section[data-smoke="{section}"] pre').first
                    expect(pane).to_be_visible(timeout = 15_000)
                    text = pane.inner_text()
                    info(f"{section} text: {text!r}")
                    assert (
                        text == "file.txt\nerror"
                    ), f"{section} rendered unexpected text: {text!r}"
                    assert ESC not in text, f"{section} still contains ESC"
                    assert "[32m" not in text, f"{section} still shows SGR garbage"
            except Exception:
                dump(page, vite)
                raise

            info("all production panes rendered clean text (no ANSI escapes)")
            browser.close()
    finally:
        if vite is not None:
            stop_process(vite)


if __name__ == "__main__":
    main()
