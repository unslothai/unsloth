# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scratch driver for playwright_shiki_retention.py: one reply, then the counters over time."""

from __future__ import annotations

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
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5392"))
BASE = f"http://127.0.0.1:{PORT}"

vite = start_vite(PORT)
try:
    wait_for_smoke_page(
        f"{BASE}/smoke-shiki-retention.html",
        "/smoke-shiki-retention-main.tsx",
        proc = vite,
        info = print,
    )
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args = chromium_launch_args(), headless = True)
        page = browser.new_page()
        page.on("console", lambda m: print(f"CONSOLE {m.type}: {m.text[:400]}", flush = True))
        page.on("pageerror", lambda e: print(f"PAGEERROR: {e}", flush = True))
        page.goto(f"{BASE}/smoke-shiki-retention.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => window.__sd && window.__sd.ready", timeout = 120_000)
        print("ready", flush = True)
        result = page.evaluate(
            "(spec) => window.__sd.runOne(spec)",
            {"kind": "stream", "chars": 8192, "seed": 1, "ticks": 8, "tickMs": 20},
        )
        print("runOne ->", result, flush = True)
        for _ in range(20):
            time.sleep(0.35)
            print(page.evaluate("() => window.__sd.counters()"), flush = True)
        browser.close()
finally:
    stop_process(vite)
