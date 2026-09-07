# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Check composer SVG and glyph centers across browsers and display settings.

Run after npm ci and Playwright browser installation:
    python tests/studio/playwright_composer_icons.py chromium webkit firefox

Measures CSS geometry; screen antialiasing may differ.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

from _playwright_robust import start_vite, stop_process, wait_for_smoke_page


MEASURE = """() => [...document.querySelectorAll('button[data-case]')].map(button => {
    const svg = button.querySelector('svg');
    const buttonBox = button.getBoundingClientRect();
    const svgBox = svg.getBoundingClientRect();
    const glyph = svg.getBBox();
    const glyphCenter = new DOMPoint(glyph.x + glyph.width / 2, glyph.y + glyph.height / 2)
        .matrixTransform(svg.getScreenCTM());
    const centerX = buttonBox.x + buttonBox.width / 2;
    const centerY = buttonBox.y + buttonBox.height / 2;
    return {
        name: button.dataset.case,
        svgDx: svgBox.x + svgBox.width / 2 - centerX,
        svgDy: svgBox.y + svgBox.height / 2 - centerY,
        glyphDx: glyphCenter.x - centerX,
        glyphDy: glyphCenter.y - centerY,
        width: svgBox.width,
        height: svgBox.height,
    };
})"""


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("engines", nargs = "*", default = ["chromium"])
    parser.add_argument("--out", type = Path, default = Path("logs/playwright-composer-icons"))
    args = parser.parse_args()
    args.out.mkdir(parents = True, exist_ok = True)
    port = int(os.environ.get("PW_PORT", "5417"))
    server = start_vite(port)
    results = []
    failures = []
    try:
        wait_for_smoke_page(
            f"http://127.0.0.1:{port}/smoke-composer-icons.html",
            "/smoke-composer-icons-main.tsx",
            proc = server,
        )
        with sync_playwright() as pw:
            for engine in args.engines:
                browser = getattr(pw, engine).launch()
                try:
                    for dpr in (1, 1.25, 1.5, 1.75, 2, 3):
                        context = browser.new_context(device_scale_factor = dpr)
                        try:
                            page = context.new_page()
                            page.goto(f"http://127.0.0.1:{port}/smoke-composer-icons.html")
                            page.locator("button[data-case]").last.wait_for()
                            for font, zoom, dark, direction in itertools.product(
                                (0.75, 1, 1.25), (0.8, 1, 1.25), (False, True), ("ltr", "rtl")
                            ):
                                case = dict(
                                    engine = engine,
                                    dpr = dpr,
                                    font = font,
                                    zoom = zoom,
                                    dark = dark,
                                    direction = direction,
                                )
                                page.evaluate(
                                    """async ({font, zoom, dark, direction}) => {
                                    const root = document.documentElement;
                                    root.style.setProperty('--ui-font-scale', String(font));
                                    root.style.zoom = String(zoom);
                                    root.classList.toggle('dark', dark);
                                    root.dir = direction;
                                    await new Promise(requestAnimationFrame);
                                }""",
                                    case,
                                )
                                measures = page.evaluate(MEASURE)
                                assert len(measures) == 6, measures
                                results.append(dict(case = case, measures = measures))
                                for measure in measures:
                                    offsets = [
                                        measure[key]
                                        for key in ("svgDx", "svgDy", "glyphDx", "glyphDy")
                                    ]
                                    # Allow layout rounding, but reject the old pixel offsets.
                                    if (
                                        max(map(abs, offsets)) > 0.02
                                        or min(measure["width"], measure["height"]) <= 0
                                    ):
                                        failures.append(dict(case = case, measure = measure))
                                if font == zoom == 1 and direction == "ltr":
                                    page.screenshot(
                                        path = str(
                                            args.out
                                            / f"{engine}-{dpr}-{'dark' if dark else 'light'}.png"
                                        )
                                    )
                        finally:
                            context.close()
                    print(f"{engine}: checked 216 configurations / 1296 icons", flush = True)
                finally:
                    browser.close()
    finally:
        stop_process(server)
        (args.out / "report.json").write_text(
            json.dumps(dict(results = results, failures = failures), indent = 2), encoding = "utf-8"
        )
    assert not failures, json.dumps(failures[:8], indent = 2)
    print(f"PASS: {len(results) * 6} rendered icons centered", flush = True)


if __name__ == "__main__":
    main()
