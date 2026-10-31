# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checks whether code that looks like a link definition strips a reply's code-block controls."""

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5231"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
PAGE = "smoke-link-definition-probe.html"
LABEL = os.environ.get("SMOKE_LABEL", "tree")

CASES = ("code", "link", "plain")

EXPECTED_FENCES = {"code": 2, "link": 1, "plain": 1}


def info(message: str) -> None:
    print(message, flush = True)


def warm_up(page) -> None:
    """Waits for the lazy Shiki and code action bar chunk, since a quiet interval reads unloaded as zero."""
    page.goto(f"{BASE}/{PAGE}?case=plain", wait_until = "domcontentloaded")
    page.wait_for_function("() => window.__probe && window.__probe.ready()", timeout = 60_000)
    page.wait_for_function(
        "() => window.__probe.counts().copyButtons >= 1",
        timeout = 120_000,
        polling = 250,
    )


def run_case(page, case: str) -> dict:
    page.goto(f"{BASE}/{PAGE}?case={case}", wait_until = "domcontentloaded")
    page.wait_for_function("() => window.__probe && window.__probe.ready()", timeout = 60_000)

    # Do not wait on the button count: `code` is legitimately zero before the fix.
    expected = EXPECTED_FENCES[case]
    page.wait_for_function(
        f"() => window.__probe.counts().codeBlocks === {expected}",
        timeout = 60_000,
        polling = 250,
    )
    page.wait_for_function(
        """() => {
            const now = JSON.stringify(window.__probe.counts());
            const seen = window.__settled;
            window.__settled = now === seen?.value
                ? {value: now, hits: seen.hits + 1}
                : {value: now, hits: 0};
            return window.__settled.hits >= 3;
        }""",
        timeout = 60_000,
        polling = 250,
    )
    return page.evaluate("() => window.__probe.counts()")


def main() -> int:
    proc = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        proc = start_vite(PORT)
    wait_for_smoke_page(
        f"{BASE}/{PAGE}", "smoke-link-definition-probe-main.tsx", proc = proc, info = info
    )
    results = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = chromium_launch_args())
            page = browser.new_page(viewport = {"width": 1280, "height": 900})
            warm_up(page)
            for case in CASES:
                results[case] = run_case(page, case)
                info(f"{case}: {json.dumps(results[case])}")
            browser.close()
    finally:
        if proc is not None:
            stop_process(proc)
            info("vite stopped")

    print(json.dumps({"label": LABEL, "results": results}, indent = 2))

    failures = []
    plain = results["plain"]
    if plain["copyButtons"] < 1 or plain["downloadButtons"] < 1:
        failures.append(
            "the `plain` control has no code-block controls, so the fixture measured nothing "
            "and no other row means anything"
        )
    code = results["code"]
    if code["copyButtons"] != EXPECTED_FENCES["code"]:
        failures.append(
            f"`code` mounted {code['copyButtons']} Copy code buttons, expected "
            f"{EXPECTED_FENCES['code']}: ordinary code took the document render path"
        )
    if code["downloadButtons"] != EXPECTED_FENCES["code"]:
        failures.append(
            f"`code` mounted {code['downloadButtons']} Download file buttons, expected "
            f"{EXPECTED_FENCES['code']}"
        )
    link = results["link"]
    if link["renderedLinks"] < 2:
        failures.append(
            f"`link` resolved {link['renderedLinks']} anchors, expected 2: a real reference "
            "link stopped resolving"
        )

    for line in failures:
        info("FAIL: " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
