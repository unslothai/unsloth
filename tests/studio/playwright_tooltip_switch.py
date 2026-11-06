# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Moving between adjacent icon buttons must show the label of the icon under the pointer (#13644).

Run after npm ci and Playwright browser installation:
    python tests/studio/playwright_tooltip_switch.py chromium firefox webkit

Hovers each icon of the assistant action bar, moves to a neighbour, and reads the open tooltip.
"""

from __future__ import annotations

import argparse
import json
import os

from playwright.sync_api import sync_playwright

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

NAV_TIMEOUT_MS = 90_000
PAGE = "smoke-tooltip-switch.html"
# Points inside the round button; its bounding-box corners are not on the button.
OFFSETS = [(dx, dy) for dx in (-0.3, 0, 0.3) for dy in (-0.3, 0, 0.3)]

OPEN_LABELS = """() => [...document.querySelectorAll('[data-slot="tooltip-content"]')]
    .map(el => el.firstChild ? el.firstChild.textContent : el.textContent)"""


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("engines", nargs = "*", default = ["chromium"])
    args = parser.parse_args()
    port = int(os.environ.get("PW_PORT", "5433"))
    server = start_vite(port)
    failures = []
    checked = 0
    try:
        url = f"http://127.0.0.1:{port}/{PAGE}"
        wait_for_smoke_page(url, "/smoke-tooltip-switch-main.tsx", proc = server)
        with sync_playwright() as pw:
            for engine in args.engines:
                browser = getattr(pw, engine).launch()
                before = checked
                try:
                    page = browser.new_page(viewport = {"width": 900, "height": 400})
                    page.set_default_timeout(NAV_TIMEOUT_MS)
                    page.goto(url, wait_until = "domcontentloaded", timeout = NAV_TIMEOUT_MS)
                    page.locator("button[data-case]").last.wait_for()
                    boxes = page.eval_on_selector_all(
                        "button[data-case]",
                        """els => els.map(b => { const r = b.getBoundingClientRect();
                        return {name: b.dataset.case, x: r.x + r.width / 2,
                                y: r.y + r.height / 2, size: r.width}; })""",
                    )
                    for i, source in enumerate(boxes):
                        for target in (boxes[j] for j in (i - 1, i + 1) if 0 <= j < len(boxes)):
                            for dx, dy in OFFSETS:
                                page.mouse.move(5, 390)
                                page.wait_for_timeout(100)
                                page.mouse.move(source["x"], source["y"], steps = 3)
                                page.wait_for_timeout(100)
                                page.mouse.move(
                                    target["x"] + dx * target["size"],
                                    target["y"] + dy * target["size"],
                                    steps = 4,
                                )
                                page.wait_for_timeout(120)
                                shown = page.evaluate(OPEN_LABELS)
                                checked += 1
                                if shown != [target["name"]]:
                                    failures.append(
                                        dict(
                                            engine = engine,
                                            source = source["name"],
                                            target = target["name"],
                                            offset = [dx, dy],
                                            shown = shown,
                                        )
                                    )
                    print(f"{engine}: checked {checked - before} moves", flush = True)
                finally:
                    browser.close()
    finally:
        stop_process(server)
    assert checked, "no moves checked"
    assert not failures, f"{len(failures)}/{checked} moves showed the wrong label:\n" + json.dumps(
        failures[:8], indent = 2
    )
    print(f"PASS: {checked} moves showed the hovered icon's label", flush = True)


if __name__ == "__main__":
    main()
