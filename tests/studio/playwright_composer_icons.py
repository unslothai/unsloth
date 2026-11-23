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


# The first navigation compiles the whole vite module graph and exceeded 30s on cold Windows runners.
NAV_TIMEOUT_MS = 90_000

MEASURE = """() => [...document.querySelectorAll('button[data-case]')].map(button => {
    const svg = button.querySelector('svg');
    const buttonBox = button.getBoundingClientRect();
    const svgBox = svg.getBoundingClientRect();
    const glyph = svg.getBBox();
    // Scale from the rendered box, not from getScreenCTM(). Firefox leaves CSS `zoom` out of
    // the CTM while getBoundingClientRect() applies it, so a CTM-mapped centre lands 1/zoom
    // out and reads as a misaligned glyph on a correctly aligned icon.
    const view = svg.viewBox.baseVal;
    const scaleX = view.width ? svgBox.width / view.width : 1;
    const scaleY = view.height ? svgBox.height / view.height : 1;
    const glyphCenter = {
        x: svgBox.x + (glyph.x + glyph.width / 2 - view.x) * scaleX,
        y: svgBox.y + (glyph.y + glyph.height / 2 - view.y) * scaleY,
    };
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

# Applies one display case and measures it in the same evaluate, so a reload cannot reset the case
# between the two and have the default layout recorded under the case's name.
APPLY_AND_MEASURE = (
    """async ({font, zoom, dark, direction}) => {
    const root = document.documentElement;
    root.style.setProperty('--ui-font-scale', String(font));
    root.style.zoom = String(zoom);
    root.classList.toggle('dark', dark);
    root.dir = direction;
    await new Promise(requestAnimationFrame);
    return ("""
    + MEASURE
    + """)();
}"""
)

CASE_ATTEMPTS = 3


def measure_case(
    page,
    case: dict,
    server = None,
) -> list:
    """Measures a case, re-running it on a fresh render when the page navigates away mid-case.

    The vite dev page sometimes reloads itself on Windows runners after its buttons have rendered
    (main job 119495424607, PR 14592 job 119703801606); the evaluate then loses its context."""
    for attempt in range(1, CASE_ATTEMPTS + 1):
        try:
            return page.evaluate(APPLY_AND_MEASURE, case)
        except Exception as exc:
            if (
                attempt == CASE_ATTEMPTS
                or "execution context was destroyed" not in str(exc).lower()
            ):
                raise
            tail = "\n".join(getattr(server, "vite_tail", []))
            print(
                f"page navigated during {case} (attempt {attempt}); re-measuring on the new render\n"
                f"vite output:\n{tail}",
                flush = True,
            )
            page.wait_for_load_state("domcontentloaded")
            page.locator("button[data-case]").last.wait_for()


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
                            page.set_default_timeout(NAV_TIMEOUT_MS)
                            page.goto(
                                f"http://127.0.0.1:{port}/smoke-composer-icons.html",
                                wait_until = "domcontentloaded",
                                timeout = NAV_TIMEOUT_MS,
                            )
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
                                measures = measure_case(page, case, server)
                                assert len(measures) == 6, measures
                                results.append(dict(case = case, measures = measures))
                                for measure in measures:
                                    offsets = [
                                        measure[key]
                                        for key in ("svgDx", "svgDy", "glyphDx", "glyphDy")
                                    ]
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
